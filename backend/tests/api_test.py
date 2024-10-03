from test_user import main_function_test_user
from test_mobileapp import test_update_details_mobile_app
from test_media import test_get_media_allowed_extensions, test_upload_get_delete_media_files
from test_notify import main_function_test_notify
from test_work import main_function_test_work

from settings.configs import HOST

if __name__ == "__main__":
    main_function_test_user()
    test_update_details_mobile_app()
    main_function_test_notify()
    main_function_test_work()

    # # TODO : Fix on production CPU Overload if upload media.
    # if HOST != "api.qfreethailand.com":
    #     """
    #         Route Media is long time to test, recomment let's last test
    #         And if you want to test only Media, you can comment all test above
    #         Or if you want to test others route, you can comment this test
    #     """
    #     test_get_media_allowed_extensions()
    #     test_upload_get_delete_media_files()